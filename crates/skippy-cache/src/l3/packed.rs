//! Append-only physical storage for logical L3 segments.
//!
//! One spill publishes at most one immutable pack. A local sidecar maps the
//! manifest's portable segment digests to pack offsets, keeping the handoff
//! manifest independent of this node's physical layout.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Mutex, RwLock},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::{segment_digest, tempfile_in, write_atomically};
use crate::fsinfo;

pub(super) const PACK_DIR: &str = "packs";
pub(super) const PACK_INDEX_DIR: &str = "pack-indexes";
const PACK_INDEX_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(super) struct PackedSegmentLocation {
    pub pack_digest: String,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct PackedManifestIndex {
    version: u32,
    payload_digest: String,
    segments: Vec<Option<PackedSegmentLocation>>,
}

#[derive(Debug)]
pub(super) struct PackedStoredSegment {
    pub new: bool,
}

#[derive(Debug)]
pub(super) struct PackedReadError {
    pub pack_digest: String,
    pub error: anyhow::Error,
}

#[derive(Debug)]
pub(super) struct PackedReadRequest<'a> {
    pub digest: &'a str,
    pub bytes: u64,
    pub location: &'a PackedSegmentLocation,
    pub output_offset: u64,
}

#[derive(Debug)]
pub(super) struct PackedSegmentStore {
    directory: PathBuf,
    index_directory: PathBuf,
    locations: RwLock<HashMap<String, PackedSegmentLocation>>,
    /// Serializes immutable pack publication with orphan collection.
    mutation: Mutex<()>,
}

impl PackedSegmentStore {
    pub(super) fn open(root: &Path) -> Result<Self> {
        let directory = root.join(PACK_DIR);
        let index_directory = root.join(PACK_INDEX_DIR);
        for path in [&directory, &index_directory] {
            fsinfo::refuse_symlinked_descendant(root, path)?;
            fs::create_dir_all(path)
                .with_context(|| format!("failed to create {}", path.display()))?;
            fsinfo::restrict_to_owner(path, 0o700)?;
        }
        Ok(Self {
            directory,
            index_directory,
            locations: RwLock::new(HashMap::new()),
            mutation: Mutex::new(()),
        })
    }

    pub(super) fn pack_path(&self, pack_digest: &str) -> PathBuf {
        self.directory.join(format!("{pack_digest}.pack"))
    }

    pub(super) fn index_path(&self, payload_digest: &str) -> PathBuf {
        self.index_directory.join(format!("{payload_digest}.json"))
    }

    pub(super) fn estimated_new_bytes(&self, segments: &[(&str, &[u8])]) -> u64 {
        let locations = self
            .locations
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut unique = HashSet::new();
        segments
            .iter()
            .filter(|(digest, _)| unique.insert(*digest))
            .filter(|(digest, _)| {
                locations
                    .get(*digest)
                    .is_none_or(|location| !self.location_is_present(location))
            })
            .map(|(_, bytes)| bytes.len() as u64)
            .fold(0u64, u64::saturating_add)
    }

    /// Publish all missing logical segments as one immutable pack.
    pub(super) fn write_batch(
        &self,
        segments: &[(&str, &[u8])],
    ) -> Result<(Vec<PackedStoredSegment>, u64)> {
        let _mutation = self
            .mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let existing = self
            .locations
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let mut planned = HashMap::<String, PackedSegmentLocation>::new();
        let mut pack_segments = Vec::new();
        let mut pack_hasher = blake3::Hasher::new();
        let mut candidate_bytes = 0u64;
        let mut new_digests = HashSet::new();

        for (digest, bytes) in segments {
            if existing
                .get(*digest)
                .is_some_and(|location| self.location_is_present(location))
                || planned.contains_key(*digest)
            {
                continue;
            }
            let offset = candidate_bytes;
            candidate_bytes = candidate_bytes
                .checked_add(bytes.len() as u64)
                .context("packed object size overflows u64")?;
            pack_hasher.update(bytes);
            pack_segments.push(*bytes);
            planned.insert(
                (*digest).to_string(),
                PackedSegmentLocation {
                    pack_digest: String::new(),
                    offset,
                    bytes: bytes.len() as u64,
                },
            );
            new_digests.insert((*digest).to_string());
        }

        let mut new_bytes = 0u64;
        if !pack_segments.is_empty() {
            let pack_digest = pack_hasher.finalize().to_hex().to_string();
            let path = self.pack_path(&pack_digest);
            if path.exists() {
                let actual = fs::metadata(&path)?.len();
                if actual != candidate_bytes {
                    bail!(
                        "packed object {pack_digest} has {actual} bytes but the write has {candidate_bytes}"
                    );
                }
            } else {
                write_pack_atomically(&path, &pack_segments)?;
                fsinfo::restrict_to_owner(&path, 0o600)?;
                new_bytes = candidate_bytes;
            }
            for location in planned.values_mut() {
                location.pack_digest.clone_from(&pack_digest);
            }
        }

        let mut locations = self
            .locations
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (digest, location) in &planned {
            locations.insert(digest.clone(), location.clone());
        }
        let published_new_pack = new_bytes > 0;
        let mut reported_new = HashSet::new();
        let stored = segments
            .iter()
            .map(|(digest, _)| PackedStoredSegment {
                new: published_new_pack
                    && new_digests.contains(*digest)
                    && reported_new.insert(*digest),
            })
            .collect();
        Ok((stored, new_bytes))
    }

    pub(super) fn location(&self, digest: &str) -> Option<PackedSegmentLocation> {
        self.locations
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(digest)
            .filter(|location| self.location_is_present(location))
            .cloned()
    }

    pub(super) fn encode_manifest_index(
        &self,
        payload_digest: &str,
        segment_digests: &[String],
    ) -> Result<Option<Vec<u8>>> {
        let segments = segment_digests
            .iter()
            .map(|digest| self.location(digest))
            .collect::<Vec<_>>();
        if segments.iter().all(Option::is_none) {
            return Ok(None);
        }
        serde_json::to_vec(&PackedManifestIndex {
            version: PACK_INDEX_VERSION,
            payload_digest: payload_digest.to_string(),
            segments,
        })
        .context("failed to serialize packed manifest index")
        .map(Some)
    }

    pub(super) fn publish_manifest_index(
        &self,
        payload_digest: &str,
        encoded: &[u8],
    ) -> Result<()> {
        let path = self.index_path(payload_digest);
        write_atomically(&path, encoded)?;
        fsinfo::restrict_to_owner(&path, 0o600)
    }

    pub(super) fn load_manifest_index(
        &self,
        payload_digest: &str,
        segment_count: usize,
    ) -> Result<Vec<Option<PackedSegmentLocation>>> {
        let path = self.index_path(payload_digest);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(vec![None; segment_count]);
            }
            Err(error) => return Err(error.into()),
        };
        let index: PackedManifestIndex =
            serde_json::from_slice(&bytes).context("malformed packed manifest index")?;
        if index.version != PACK_INDEX_VERSION
            || index.payload_digest != payload_digest
            || index.segments.len() != segment_count
        {
            bail!("packed manifest index does not match manifest {payload_digest}");
        }
        for location in index.segments.iter().flatten() {
            if !is_digest(&location.pack_digest) || location.bytes == 0 {
                bail!("packed manifest index contains an invalid location");
            }
            location
                .offset
                .checked_add(location.bytes)
                .context("packed manifest index range overflows")?;
        }
        Ok(index.segments)
    }

    pub(super) fn remove_manifest_index(&self, payload_digest: &str) -> Result<u64> {
        let path = self.index_path(payload_digest);
        let bytes = fs::metadata(&path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        match fs::remove_file(&path) {
            Ok(()) => Ok(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(error) => Err(error.into()),
        }
    }

    pub(super) fn validate(&self, location: &PackedSegmentLocation, bytes: u64) -> Result<()> {
        if location.bytes != bytes {
            bail!(
                "packed index records {} bytes but manifest records {bytes}",
                location.bytes
            );
        }
        let pack_bytes = fs::metadata(self.pack_path(&location.pack_digest))
            .with_context(|| format!("missing pack {}", location.pack_digest))?
            .len();
        let end = location
            .offset
            .checked_add(bytes)
            .context("packed segment range overflows")?;
        if end > pack_bytes {
            bail!(
                "pack {} has {pack_bytes} bytes but segment range ends at {end}",
                location.pack_digest
            );
        }
        Ok(())
    }

    /// Append requests directly into their final payload ranges.
    ///
    /// Consecutive logical segments that are also consecutive in one pack are
    /// issued as one read. Digest verification still happens per logical
    /// segment, preserving the manifest contract without allocating one
    /// temporary `Vec` for every segment and then copying it into the payload.
    pub(super) fn append_many(
        &self,
        requests: &[PackedReadRequest<'_>],
        output: &mut Vec<u8>,
    ) -> Result<(), PackedReadError> {
        let mut files = HashMap::<String, File>::new();
        let mut first = 0usize;
        while first < requests.len() {
            let first_request = &requests[first];
            let mut last = first + 1;
            let mut physical_end = first_request
                .location
                .offset
                .checked_add(first_request.bytes)
                .context("packed read range overflows")
                .map_err(|error| PackedReadError {
                    pack_digest: first_request.location.pack_digest.clone(),
                    error,
                })?;
            let mut output_end = first_request
                .output_offset
                .checked_add(first_request.bytes)
                .context("packed output range overflows")
                .map_err(|error| PackedReadError {
                    pack_digest: first_request.location.pack_digest.clone(),
                    error,
                })?;
            while let Some(next) = requests.get(last) {
                if next.location.pack_digest != first_request.location.pack_digest
                    || next.location.offset != physical_end
                    || next.output_offset != output_end
                {
                    break;
                }
                physical_end =
                    physical_end
                        .checked_add(next.bytes)
                        .ok_or_else(|| PackedReadError {
                            pack_digest: first_request.location.pack_digest.clone(),
                            error: anyhow::anyhow!("packed read range overflows"),
                        })?;
                output_end = output_end
                    .checked_add(next.bytes)
                    .ok_or_else(|| PackedReadError {
                        pack_digest: first_request.location.pack_digest.clone(),
                        error: anyhow::anyhow!("packed output range overflows"),
                    })?;
                last += 1;
            }

            let result = (|| -> Result<()> {
                if usize::try_from(first_request.output_offset)
                    .context("packed output offset exceeds usize")?
                    != output.len()
                {
                    bail!("packed output range is not contiguous with the payload");
                }
                let file = match files.entry(first_request.location.pack_digest.clone()) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let file = File::open(self.pack_path(&first_request.location.pack_digest))
                            .with_context(|| {
                                format!(
                                    "failed to open pack {}",
                                    first_request.location.pack_digest
                                )
                            })?;
                        entry.insert(file)
                    }
                };
                let output_end =
                    usize::try_from(output_end).context("packed output end exceeds usize")?;
                file.seek(SeekFrom::Start(first_request.location.offset))?;
                let run_bytes = output_end
                    .checked_sub(output.len())
                    .context("packed output range precedes payload")?;
                let read = file
                    .take(run_bytes as u64)
                    .read_to_end(output)
                    .context("failed to read packed payload range")?;
                if read != run_bytes {
                    bail!("packed payload range ended after {read} of {run_bytes} bytes");
                }

                for request in &requests[first..last] {
                    let start = usize::try_from(request.output_offset)
                        .context("segment output offset exceeds usize")?;
                    let end = usize::try_from(
                        request
                            .output_offset
                            .checked_add(request.bytes)
                            .context("segment output range overflows")?,
                    )
                    .context("segment output end exceeds usize")?;
                    let bytes = output
                        .get(start..end)
                        .context("segment output range exceeds payload")?;
                    if segment_digest(bytes) != request.digest {
                        bail!(
                            "packed segment {} failed digest verification",
                            request.digest
                        );
                    }
                }
                Ok(())
            })();
            match result {
                Ok(()) => {}
                Err(error) => {
                    return Err(PackedReadError {
                        pack_digest: first_request.location.pack_digest.clone(),
                        error,
                    });
                }
            }
            first = last;
        }
        Ok(())
    }

    pub(super) fn rebuild(
        &self,
        entries: impl IntoIterator<Item = (String, PackedSegmentLocation)>,
    ) {
        let mut locations = self
            .locations
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        locations.clear();
        locations.extend(entries);
    }

    pub(super) fn remove_orphan_packs(
        &self,
        referenced_segments: &HashSet<String>,
        held_segments: &HashSet<String>,
    ) -> Result<u64> {
        let _mutation = self
            .mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let locations = self
            .locations
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let protected = locations
            .iter()
            .filter(|(digest, _)| {
                referenced_segments.contains(*digest) || held_segments.contains(*digest)
            })
            .map(|(_, location)| location.pack_digest.clone())
            .collect::<HashSet<_>>();
        let mut freed = 0u64;
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|extension| extension != "pack") {
                continue;
            }
            let Some(pack_digest) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if protected.contains(pack_digest) {
                continue;
            }
            freed = freed.saturating_add(entry.metadata()?.len());
            fs::remove_file(&path)
                .with_context(|| format!("failed to collect pack {pack_digest}"))?;
        }
        Ok(freed)
    }

    pub(super) fn remove_orphan_indexes(&self, manifests: &HashSet<String>) -> Result<u64> {
        let mut freed = 0u64;
        for entry in fs::read_dir(&self.index_directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|extension| extension != "json") {
                continue;
            }
            let Some(payload_digest) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if manifests.contains(payload_digest) {
                continue;
            }
            freed = freed.saturating_add(entry.metadata()?.len());
            fs::remove_file(&path)?;
        }
        Ok(freed)
    }

    pub(super) fn footprint_bytes(&self) -> Result<u64> {
        directory_bytes(&self.directory)
    }

    pub(super) fn index_footprint_bytes(&self) -> Result<u64> {
        directory_bytes(&self.index_directory)
    }

    fn location_is_present(&self, location: &PackedSegmentLocation) -> bool {
        self.pack_path(&location.pack_digest)
            .metadata()
            .is_ok_and(|metadata| location.offset < metadata.len())
    }
}

fn directory_bytes(directory: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            total = total.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(total)
}

fn write_pack_atomically(path: &Path, segments: &[&[u8]]) -> Result<()> {
    let directory = path.parent().context("pack path has no parent directory")?;
    let (temp_path, mut temp_file) = tempfile_in(directory)?;
    let write_result = (|| -> Result<()> {
        for segment in segments {
            temp_file
                .write_all(segment)
                .with_context(|| format!("failed to write {}", temp_path.display()))?;
        }
        temp_file
            .sync_all()
            .with_context(|| format!("failed to sync {}", temp_path.display()))
    })();
    drop(temp_file);
    let publish_result = write_result.and_then(|()| fsinfo::replace_file(&temp_path, path));
    if let Err(error) = publish_result {
        return match fs::remove_file(&temp_path) {
            Ok(()) => Err(error),
            Err(cleanup) if cleanup.kind() == std::io::ErrorKind::NotFound => Err(error),
            Err(cleanup) => Err(error.context(format!(
                "also failed to remove temporary pack {}: {cleanup}",
                temp_path.display()
            ))),
        };
    }
    Ok(())
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
