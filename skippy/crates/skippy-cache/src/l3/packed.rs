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

/// Cost accounting for one packed read pass.
///
/// Restore latency is dominated by physical reads and per-object work, not by
/// bytes, so the caller records both: how many runs were issued, how many
/// distinct packs they touched, and how long the reads themselves took.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct ReadAccounting {
    pub(super) packs_opened: u64,
    pub(super) physical_reads: u64,
    pub(super) bytes_read: u64,
    pub(super) read_nanos: u64,
    pub(super) max_read_nanos: u64,
    /// CPU time spent hashing segment digests (summed across workers).
    pub(super) verify_nanos: u64,
    /// CPU time spent hashing the assembled payload (summed across workers).
    pub(super) hash_nanos: u64,
    pub(super) hashed_bytes: u64,
}

impl ReadAccounting {
    fn record(&mut self, bytes: usize, nanos: u64) {
        self.physical_reads = self.physical_reads.saturating_add(1);
        self.bytes_read = self.bytes_read.saturating_add(bytes as u64);
        self.read_nanos = self.read_nanos.saturating_add(nanos);
        self.max_read_nanos = self.max_read_nanos.max(nanos);
    }
}

/// One physically contiguous read: consecutive manifest segments that are also
/// consecutive in the same pack.
#[derive(Debug, Clone, Copy)]
pub(super) struct ReadRun<'a> {
    pub(super) pack_digest: &'a str,
    pub(super) pack_offset: u64,
    pub(super) output_offset: u64,
    pub(super) bytes: u64,
    pub(super) first_request: usize,
    pub(super) request_end: usize,
}

/// One physical read covering requests that are adjacent inside a pack, even
/// when they are far apart in the assembled payload.
#[derive(Debug, Clone)]
struct PackRun {
    pack_digest_index: usize,
    pack_offset: u64,
    bytes: u64,
    /// Manifest indices carried by this read, in physical order.
    requests: Vec<usize>,
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
        super::manifest_codec::encode_index(PACK_INDEX_VERSION, payload_digest, &segments)
            .context("failed to encode packed manifest index")
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
        let (version, indexed_digest, segments) = if super::manifest_codec::looks_binary(&bytes) {
            super::manifest_codec::decode_index(&bytes)
                .context("malformed packed manifest index")?
        } else {
            let legacy: PackedManifestIndex =
                serde_json::from_slice(&bytes).context("malformed packed manifest index")?;
            (legacy.version, legacy.payload_digest, legacy.segments)
        };
        if version != PACK_INDEX_VERSION
            || indexed_digest != payload_digest
            || segments.len() != segment_count
        {
            bail!("packed manifest index does not match manifest {payload_digest}");
        }
        for location in segments.iter().flatten() {
            if !is_digest(&location.pack_digest) || location.bytes == 0 {
                bail!("packed manifest index contains an invalid location");
            }
            location
                .offset
                .checked_add(location.bytes)
                .context("packed manifest index range overflows")?;
        }
        Ok(segments)
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
    ///
    /// A restore that spans several packs issues their runs concurrently: the
    /// per-object cost (open, seek, submit) dominates a payload assembled from
    /// many small spill packs, and the runs write disjoint ranges.
    pub(super) fn append_many(
        &self,
        requests: &[PackedReadRequest<'_>],
        output: &mut Vec<u8>,
        mut payload_hasher: Option<&mut blake3::Hasher>,
        accounting: &mut ReadAccounting,
    ) -> Result<(), PackedReadError> {
        if requests.is_empty() {
            return Ok(());
        }
        let base = output.len();
        // One pack is the single-spill case: the manifest-adjacent path reads
        // it in one run and verifies its segments in parallel, without the
        // scatter copy a multi-pack compose needs.
        let single_pack = requests
            .iter()
            .all(|request| request.location.pack_digest == requests[0].location.pack_digest);
        if !single_pack && let Some(total) = contiguous_output_len(requests, base) {
            // The common restore: every segment is packed and the request
            // range covers the whole payload. Group by pack, not by manifest
            // adjacency, so a payload assembled from N spill packs costs N
            // reads rather than N reads per run.
            self.append_packed_major(
                requests,
                base,
                total,
                output,
                &mut payload_hasher,
                accounting,
            )?;
            return Ok(());
        }
        let runs = self.plan_runs(requests)?;
        if runs.len() == 1 {
            // The hot path: one contiguous run goes straight into the payload,
            // no scratch buffer and no second copy.
            self.read_run(&runs[0], requests, output, accounting)?;
        } else {
            let mut buffers = self.read_runs_concurrently(&runs, requests, accounting)?;
            for (run, bytes) in runs.iter().zip(buffers.drain(..)) {
                if usize::try_from(run.output_offset).ok() != Some(output.len()) {
                    return Err(PackedReadError {
                        pack_digest: run.pack_digest.to_string(),
                        error: anyhow::anyhow!(
                            "packed output range is not contiguous with the payload"
                        ),
                    });
                }
                output.extend_from_slice(&bytes);
            }
        }
        if let Some(hasher) = payload_hasher.as_mut() {
            for run in &runs {
                let start =
                    usize::try_from(run.output_offset).map_err(|error| PackedReadError {
                        pack_digest: run.pack_digest.to_string(),
                        error: error.into(),
                    })?;
                let len = usize::try_from(run.bytes).map_err(|error| PackedReadError {
                    pack_digest: run.pack_digest.to_string(),
                    error: error.into(),
                })?;
                let end = start.checked_add(len).ok_or_else(|| PackedReadError {
                    pack_digest: run.pack_digest.to_string(),
                    error: anyhow::anyhow!("run output range overflows"),
                })?;
                let bytes = output.get(start..end).ok_or_else(|| PackedReadError {
                    pack_digest: run.pack_digest.to_string(),
                    error: anyhow::anyhow!("run output range exceeds payload"),
                })?;
                let started = std::time::Instant::now();
                hasher.update(bytes);
                accounting.hash_nanos = accounting
                    .hash_nanos
                    .saturating_add(started.elapsed().as_nanos() as u64);
                accounting.hashed_bytes =
                    accounting.hashed_bytes.saturating_add(bytes.len() as u64);
            }
        }
        Ok(())
    }

    /// Group manifest-ordered requests into physically contiguous runs.
    fn plan_runs<'a>(
        &self,
        requests: &'a [PackedReadRequest<'a>],
    ) -> Result<Vec<ReadRun<'a>>, PackedReadError> {
        let mut runs = Vec::new();
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
            runs.push(ReadRun {
                pack_digest: &first_request.location.pack_digest,
                pack_offset: first_request.location.offset,
                output_offset: first_request.output_offset,
                bytes: physical_end - first_request.location.offset,
                first_request: first,
                request_end: last,
            });
            first = last;
        }
        Ok(runs)
    }

    /// Read a whole packed payload pack-first.
    ///
    /// Writes in a turn-by-turn cache are turn-major but the payload is
    /// block-major, so manifest-adjacent coalescing degenerates to one read
    /// per (pack, block) pair. Grouping by pack and coalescing on physical
    /// adjacency recovers one read per pack.
    fn append_packed_major(
        &self,
        requests: &[PackedReadRequest<'_>],
        base: usize,
        total: u64,
        output: &mut Vec<u8>,
        payload_hasher: &mut Option<&mut blake3::Hasher>,
        accounting: &mut ReadAccounting,
    ) -> Result<(), PackedReadError> {
        let total = usize::try_from(total).map_err(|error| PackedReadError {
            pack_digest: String::new(),
            error: error.into(),
        })?;
        let end = base.checked_add(total).ok_or_else(|| PackedReadError {
            pack_digest: String::new(),
            error: anyhow::anyhow!("packed payload size overflows usize"),
        })?;
        output.resize(end, 0);

        let mut packs: Vec<&str> = Vec::new();
        let mut by_pack: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, request) in requests.iter().enumerate() {
            let digest = request.location.pack_digest.as_str();
            if !by_pack.contains_key(digest) {
                packs.push(digest);
            }
            by_pack.entry(digest).or_default().push(index);
        }
        let mut runs: Vec<PackRun> = Vec::new();
        for (pack_index, digest) in packs.iter().enumerate() {
            let mut indices = by_pack.remove(*digest).unwrap_or_default();
            indices.sort_by_key(|index| requests[*index].location.offset);
            let mut first = 0usize;
            while first < indices.len() {
                let mut last = first + 1;
                let mut physical_end = requests[indices[first]]
                    .location
                    .offset
                    .checked_add(requests[indices[first]].bytes)
                    .ok_or_else(|| PackedReadError {
                        pack_digest: (*digest).to_string(),
                        error: anyhow::anyhow!("packed read range overflows"),
                    })?;
                let run_start = requests[indices[first]].location.offset;
                while let Some(next) = indices.get(last) {
                    let next_offset = requests[*next].location.offset;
                    if next_offset != physical_end {
                        // A spill leaves superseded partial-window segments
                        // behind, so referenced segments are not always
                        // physically adjacent. Reading across a small gap is
                        // cheaper than paying another open and seek, as long
                        // as the wasted bytes stay a minority of the run.
                        let gap = next_offset.saturating_sub(physical_end);
                        if next_offset < physical_end
                            || gap.saturating_mul(4) > physical_end - run_start
                        {
                            break;
                        }
                    }
                    physical_end =
                        next_offset
                            .checked_add(requests[*next].bytes)
                            .ok_or_else(|| PackedReadError {
                                pack_digest: (*digest).to_string(),
                                error: anyhow::anyhow!("packed read range overflows"),
                            })?;
                    last += 1;
                }
                runs.push(PackRun {
                    pack_digest_index: pack_index,
                    pack_offset: requests[indices[first]].location.offset,
                    bytes: physical_end - requests[indices[first]].location.offset,
                    requests: indices[first..last].to_vec(),
                });
                first = last;
            }
        }

        let workers = read_workers(runs.len());
        let mut buffers: Vec<Option<Vec<u8>>> = (0..runs.len()).map(|_| None).collect();
        if workers <= 1 {
            for (index, run) in runs.iter().enumerate() {
                let mut bytes = Vec::new();
                self.read_pack_run(
                    run,
                    packs[run.pack_digest_index],
                    requests,
                    &mut bytes,
                    accounting,
                )?;
                buffers[index] = Some(bytes);
            }
        } else {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let outcomes = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(workers);
                for _ in 0..workers {
                    let next = &next;
                    let runs = &runs;
                    let packs = &packs;
                    handles.push(scope.spawn(move || {
                        let mut local = ReadAccounting::default();
                        let mut produced: Vec<(usize, Result<Vec<u8>, PackedReadError>)> =
                            Vec::new();
                        loop {
                            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if index >= runs.len() {
                                break;
                            }
                            let run = &runs[index];
                            let mut bytes = Vec::new();
                            match self.read_pack_run(
                                run,
                                packs[run.pack_digest_index],
                                requests,
                                &mut bytes,
                                &mut local,
                            ) {
                                Ok(()) => produced.push((index, Ok(bytes))),
                                Err(error) => {
                                    produced.push((index, Err(error)));
                                    break;
                                }
                            }
                        }
                        (local, produced)
                    }));
                }
                handles
                    .into_iter()
                    .map(|handle| {
                        handle.join().unwrap_or_else(|_| {
                            (
                                ReadAccounting::default(),
                                vec![(
                                    0,
                                    Err(PackedReadError {
                                        pack_digest: String::new(),
                                        error: anyhow::anyhow!("packed read worker panicked"),
                                    }),
                                )],
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            });
            for (local, produced) in outcomes {
                merge_accounting(accounting, &local);
                for (index, outcome) in produced {
                    let bytes = outcome?;
                    buffers[index] = Some(bytes);
                }
            }
        }

        for (run, buffer) in runs.iter().zip(buffers.iter_mut()) {
            let bytes = buffer.take().ok_or_else(|| PackedReadError {
                pack_digest: packs[run.pack_digest_index].to_string(),
                error: anyhow::anyhow!("packed run was never read"),
            })?;
            for request_index in &run.requests {
                let request = &requests[*request_index];
                let len = usize::try_from(request.bytes).map_err(|error| PackedReadError {
                    pack_digest: packs[run.pack_digest_index].to_string(),
                    error: error.into(),
                })?;
                let offset = usize::try_from(request.location.offset - run.pack_offset).map_err(
                    |error| PackedReadError {
                        pack_digest: packs[run.pack_digest_index].to_string(),
                        error: error.into(),
                    },
                )?;
                let slice = bytes
                    .get(offset..offset + len)
                    .ok_or_else(|| PackedReadError {
                        pack_digest: packs[run.pack_digest_index].to_string(),
                        error: anyhow::anyhow!("packed run is shorter than its segments"),
                    })?;
                let output_offset =
                    usize::try_from(request.output_offset).map_err(|error| PackedReadError {
                        pack_digest: packs[run.pack_digest_index].to_string(),
                        error: error.into(),
                    })?;
                let start = base
                    .checked_add(output_offset)
                    .ok_or_else(|| PackedReadError {
                        pack_digest: packs[run.pack_digest_index].to_string(),
                        error: anyhow::anyhow!("packed segment offset overflows"),
                    })?;
                let destination =
                    output
                        .get_mut(start..start + len)
                        .ok_or_else(|| PackedReadError {
                            pack_digest: packs[run.pack_digest_index].to_string(),
                            error: anyhow::anyhow!("packed segment range exceeds payload"),
                        })?;
                destination.copy_from_slice(slice);
            }
        }
        if let Some(hasher) = payload_hasher.as_deref_mut() {
            let started = std::time::Instant::now();
            hasher.update(&output[base..end]);
            accounting.hash_nanos = accounting
                .hash_nanos
                .saturating_add(started.elapsed().as_nanos() as u64);
            accounting.hashed_bytes = accounting.hashed_bytes.saturating_add((end - base) as u64);
        }
        Ok(())
    }

    /// Read one physically adjacent pack range into `output`.
    fn read_pack_run(
        &self,
        run: &PackRun,
        pack_digest: &str,
        requests: &[PackedReadRequest<'_>],
        output: &mut Vec<u8>,
        accounting: &mut ReadAccounting,
    ) -> Result<(), PackedReadError> {
        let mut file =
            File::open(self.pack_path(pack_digest)).map_err(|error| PackedReadError {
                pack_digest: pack_digest.to_string(),
                error: error.into(),
            })?;
        let started = std::time::Instant::now();
        let result = file
            .seek(SeekFrom::Start(run.pack_offset))
            .and_then(|_| file.take(run.bytes).read_to_end(output));
        let elapsed = started.elapsed();
        let read = result.map_err(|error| PackedReadError {
            pack_digest: pack_digest.to_string(),
            error: error.into(),
        })?;
        let expected = usize::try_from(run.bytes).unwrap_or(usize::MAX);
        if read != expected {
            return Err(PackedReadError {
                pack_digest: pack_digest.to_string(),
                error: anyhow::anyhow!(
                    "packed payload range ended after {read} of {expected} bytes"
                ),
            });
        }
        accounting.packs_opened = accounting.packs_opened.saturating_add(1);
        accounting.record(read, elapsed.as_nanos() as u64);
        // Verify inside the worker: segment digests are independent, so
        // spreading them across the read threads keeps the compose step from
        // becoming the serial bottleneck.
        let verify_started = std::time::Instant::now();
        for request_index in &run.requests {
            let request = &requests[*request_index];
            let len = usize::try_from(request.bytes).map_err(|error| PackedReadError {
                pack_digest: pack_digest.to_string(),
                error: error.into(),
            })?;
            let start =
                usize::try_from(request.location.offset - run.pack_offset).map_err(|error| {
                    PackedReadError {
                        pack_digest: pack_digest.to_string(),
                        error: error.into(),
                    }
                })?;
            let end = start.checked_add(len).ok_or_else(|| PackedReadError {
                pack_digest: pack_digest.to_string(),
                error: anyhow::anyhow!("packed segment range overflows"),
            })?;
            let segment = output.get(start..end).ok_or_else(|| PackedReadError {
                pack_digest: pack_digest.to_string(),
                error: anyhow::anyhow!("packed run is shorter than its segments"),
            })?;
            if segment_digest(segment) != request.digest {
                return Err(PackedReadError {
                    pack_digest: pack_digest.to_string(),
                    error: anyhow::anyhow!(
                        "packed segment {} failed digest verification",
                        request.digest
                    ),
                });
            }
        }
        accounting.verify_nanos = accounting
            .verify_nanos
            .saturating_add(verify_started.elapsed().as_nanos() as u64);
        Ok(())
    }

    /// Read one run into `output`, verifying each logical segment it carries.
    fn read_run(
        &self,
        run: &ReadRun<'_>,
        requests: &[PackedReadRequest<'_>],
        output: &mut Vec<u8>,
        accounting: &mut ReadAccounting,
    ) -> Result<(), PackedReadError> {
        let path = self.pack_path(run.pack_digest);
        let mut file = File::open(&path).map_err(|error| PackedReadError {
            pack_digest: run.pack_digest.to_string(),
            error: error.into(),
        })?;
        let start = output.len();
        let started = std::time::Instant::now();
        let result = file
            .seek(SeekFrom::Start(run.pack_offset))
            .and_then(|_| file.take(run.bytes).read_to_end(output));
        let elapsed = started.elapsed();
        let read = match result {
            Ok(read) => read,
            Err(error) => {
                return Err(PackedReadError {
                    pack_digest: run.pack_digest.to_string(),
                    error: error.into(),
                });
            }
        };
        let expected = usize::try_from(run.bytes).unwrap_or(usize::MAX);
        if read != expected {
            return Err(PackedReadError {
                pack_digest: run.pack_digest.to_string(),
                error: anyhow::anyhow!(
                    "packed payload range ended after {read} of {expected} bytes"
                ),
            });
        }
        accounting.packs_opened = accounting.packs_opened.saturating_add(1);
        accounting.record(read, elapsed.as_nanos() as u64);
        let verify_started = std::time::Instant::now();
        let outcome = verify_run_segments(&output[start..start + read], run, requests);
        accounting.verify_nanos = accounting
            .verify_nanos
            .saturating_add(verify_started.elapsed().as_nanos() as u64);
        outcome
    }

    /// Read independent runs concurrently, preserving request order in the
    /// returned buffers.
    fn read_runs_concurrently(
        &self,
        runs: &[ReadRun<'_>],
        requests: &[PackedReadRequest<'_>],
        accounting: &mut ReadAccounting,
    ) -> Result<Vec<Vec<u8>>, PackedReadError> {
        // Profiling and incident-response override: force a specific read
        // concurrency instead of the machine's default.
        let configured = std::env::var("SKIPPY_L3_READ_WORKERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0);
        let workers = configured
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|value| value.get())
                    .unwrap_or(1)
            })
            .min(runs.len())
            .max(1);
        if workers == 1 {
            let mut buffers = Vec::with_capacity(runs.len());
            for run in runs {
                let mut bytes = Vec::new();
                self.read_run(run, requests, &mut bytes, accounting)?;
                buffers.push(bytes);
            }
            return Ok(buffers);
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let outcomes = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let next = &next;
                handles.push(scope.spawn(move || {
                    let mut local = ReadAccounting::default();
                    let mut produced: Vec<(usize, Result<Vec<u8>, PackedReadError>)> = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if index >= runs.len() {
                            break;
                        }
                        let mut bytes = Vec::new();
                        match self.read_run(&runs[index], requests, &mut bytes, &mut local) {
                            Ok(()) => produced.push((index, Ok(bytes))),
                            Err(error) => {
                                produced.push((index, Err(error)));
                                break;
                            }
                        }
                    }
                    (local, produced)
                }));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|_| {
                        (
                            ReadAccounting::default(),
                            vec![(
                                0,
                                Err(PackedReadError {
                                    pack_digest: String::new(),
                                    error: anyhow::anyhow!("packed read worker panicked"),
                                }),
                            )],
                        )
                    })
                })
                .collect::<Vec<_>>()
        });
        let mut results: Vec<Option<Result<Vec<u8>, PackedReadError>>> =
            (0..runs.len()).map(|_| None).collect();
        for (local, produced) in outcomes {
            accounting.packs_opened = accounting.packs_opened.saturating_add(local.packs_opened);
            accounting.physical_reads = accounting
                .physical_reads
                .saturating_add(local.physical_reads);
            accounting.bytes_read = accounting.bytes_read.saturating_add(local.bytes_read);
            accounting.read_nanos = accounting.read_nanos.saturating_add(local.read_nanos);
            accounting.max_read_nanos = accounting.max_read_nanos.max(local.max_read_nanos);
            for (index, outcome) in produced {
                if let Some(slot) = results.get_mut(index) {
                    *slot = Some(outcome);
                }
            }
        }
        results
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                result.unwrap_or_else(|| {
                    Err(PackedReadError {
                        pack_digest: runs[index].pack_digest.to_string(),
                        error: anyhow::anyhow!("packed run was never read"),
                    })
                })
            })
            .collect()
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
        referenced_packs: &HashSet<String>,
        held_segments: impl FnOnce() -> HashSet<String>,
    ) -> Result<u64> {
        let _mutation = self
            .mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Snapshot in-flight publications only after taking the same lock that
        // serializes pack publication, so collection cannot race a completed
        // pack into existence before its manifest is committed.
        let held_segments = held_segments();
        let locations = self
            .locations
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut protected = referenced_packs.clone();
        for digest in &held_segments {
            if let Some(location) = locations.get(digest) {
                protected.insert(location.pack_digest.clone());
            }
        }
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

/// Join a verification worker, treating a panic as a failed verification.
///
/// A panicked worker has checked none of its segments. Dropping it would
/// report the run as verified, so the panic is surfaced as the error it is.
fn join_verification(
    handle: std::thread::ScopedJoinHandle<'_, Result<(), PackedReadError>>,
    pack_digest: &str,
) -> Result<(), PackedReadError> {
    handle.join().unwrap_or_else(|_| {
        Err(PackedReadError {
            pack_digest: pack_digest.to_string(),
            error: anyhow::anyhow!("packed verification worker panicked"),
        })
    })
}

/// Total bytes covered by `requests` if they tile the output contiguously from
/// `base`, in manifest order.
fn contiguous_output_len(requests: &[PackedReadRequest<'_>], base: usize) -> Option<u64> {
    let mut expected = u64::try_from(base).ok()?;
    for request in requests {
        if request.output_offset != expected {
            return None;
        }
        expected = expected.checked_add(request.bytes)?;
    }
    Some(expected - u64::try_from(base).ok()?)
}

/// Read concurrency: the machine's default, overridable for profiling.
fn read_workers(runs: usize) -> usize {
    let configured = std::env::var("SKIPPY_L3_READ_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0);
    configured
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1)
        })
        .min(runs)
        .max(1)
}

fn merge_accounting(into: &mut ReadAccounting, from: &ReadAccounting) {
    into.packs_opened = into.packs_opened.saturating_add(from.packs_opened);
    into.physical_reads = into.physical_reads.saturating_add(from.physical_reads);
    into.bytes_read = into.bytes_read.saturating_add(from.bytes_read);
    into.read_nanos = into.read_nanos.saturating_add(from.read_nanos);
    into.max_read_nanos = into.max_read_nanos.max(from.max_read_nanos);
    into.verify_nanos = into.verify_nanos.saturating_add(from.verify_nanos);
    into.hash_nanos = into.hash_nanos.saturating_add(from.hash_nanos);
    into.hashed_bytes = into.hashed_bytes.saturating_add(from.hashed_bytes);
}

/// Verify every logical segment carried by one physical run against its
/// manifest digest.
///
/// Segment digests are independent, so a single large run (the common case:
/// one pack holding a whole payload) splits its verification across threads
/// instead of hashing the payload serially.
fn verify_run_segments(
    bytes: &[u8],
    run: &ReadRun<'_>,
    requests: &[PackedReadRequest<'_>],
) -> Result<(), PackedReadError> {
    let count = run.request_end - run.first_request;
    if count == 0 {
        return Ok(());
    }
    let mut offsets = Vec::with_capacity(count + 1);
    let mut cursor = 0usize;
    offsets.push(0usize);
    for request in &requests[run.first_request..run.request_end] {
        let len = usize::try_from(request.bytes).map_err(|error| PackedReadError {
            pack_digest: run.pack_digest.to_string(),
            error: error.into(),
        })?;
        cursor = cursor.checked_add(len).ok_or_else(|| PackedReadError {
            pack_digest: run.pack_digest.to_string(),
            error: anyhow::anyhow!("packed segment range overflows"),
        })?;
        offsets.push(cursor);
    }
    let workers = read_workers(count);
    if workers <= 1 {
        return verify_segments_serial(bytes, run, requests, &offsets);
    }
    let chunk = count.div_ceil(workers);
    let outcomes = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk_index in 0..workers {
            let start = chunk_index * chunk;
            if start >= count {
                break;
            }
            let end = (start + chunk).min(count);
            let offsets = &offsets;
            handles.push(scope.spawn(move || {
                for index in start..end {
                    let request = &requests[run.first_request + index];
                    let segment = &bytes[offsets[index]..offsets[index + 1]];
                    if segment_digest(segment) != request.digest {
                        return Err(PackedReadError {
                            pack_digest: run.pack_digest.to_string(),
                            error: anyhow::anyhow!(
                                "packed segment {} failed digest verification",
                                request.digest
                            ),
                        });
                    }
                }
                Ok(())
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                // A panicked verifier has checked nothing: surface it as a
                // verification failure rather than dropping the worker and
                // reporting the run as verified.
                join_verification(handle, run.pack_digest)
            })
            .collect::<Vec<Result<(), PackedReadError>>>()
    });
    outcomes
        .into_iter()
        .find_map(Result::err)
        .map_or(Ok(()), Err)
}

fn verify_segments_serial(
    bytes: &[u8],
    run: &ReadRun<'_>,
    requests: &[PackedReadRequest<'_>],
    offsets: &[usize],
) -> Result<(), PackedReadError> {
    for (index, request) in requests[run.first_request..run.request_end]
        .iter()
        .enumerate()
    {
        if segment_digest(&bytes[offsets[index]..offsets[index + 1]]) != request.digest {
            return Err(PackedReadError {
                pack_digest: run.pack_digest.to_string(),
                error: anyhow::anyhow!(
                    "packed segment {} failed digest verification",
                    request.digest
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panicked_verification_worker_fails_the_run() {
        // A worker that dies has verified none of its segments; the join must
        // report a failure rather than silently dropping it.
        std::thread::scope(|scope| {
            let handle = scope.spawn(|| -> Result<(), PackedReadError> {
                panic!("verification worker died");
            });
            let outcome = join_verification(handle, "blake3:pack");
            assert!(outcome.is_err());
        });
    }

    #[test]
    fn a_completed_verification_worker_keeps_its_result() {
        std::thread::scope(|scope| {
            let handle = scope.spawn(|| Ok(()));
            assert!(join_verification(handle, "blake3:pack").is_ok());
        });
    }
}
