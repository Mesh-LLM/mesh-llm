//! A local copy of a remote GGUF that holds only the bytes an audit read.
//!
//! The copy is a file of the remote file's size. The ranges it holds are
//! fetched on first use and recorded next to it, and the rest stays a hole,
//! so it takes no disk space. The glue check loads a layer from the copy as
//! from a whole GGUF: it never reads the weight matrices, whose opened
//! outputs replace them, and the holes read as zeros.
//!
//! The record names the remote file by its pinned revision and size, and
//! exists from the moment the copy does. A range enters the record only
//! after its bytes are synced to disk, so a crash never leaves a record of
//! bytes that read as zeros.
//!
//! Every opener holds a shared lock on a lock file next to the copy. A copy
//! starts over (its record names another file, or is gone or unreadable, or
//! the data file is gone) only under the exclusive lock, which no opener
//! gets while another has the copy open: bytes another opener is about to
//! record are never truncated. Openers of one copy write the same bytes,
//! and each adds the ranges the others recorded to its own when it saves.
//!
//! The copies under one root stay within a size: [`eviction`] removes the
//! ones used longest ago that no opener holds.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::weights::ByteSource;

pub mod eviction;

/// Ranges at most this far apart are fetched as one.
const MERGE_GAP: u64 = 64 << 10;
/// The largest range one request fetches.
const MAX_FETCH: u64 = 8 << 20;
/// Requests in flight at once.
const FETCHERS: usize = 16;

/// Reads byte ranges of the remote file.
pub trait RangeFetch: Send + Sync {
    /// The bytes `[offset, offset + len)`, all of them.
    fn fetch(&self, offset: u64, len: u64) -> Result<Vec<u8>>;
}

/// What the record says the copy is a copy of.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteIdentity {
    /// Where the file comes from, pinned: for Hugging Face,
    /// `repo@commit/file`.
    pub source: String,
    pub size: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Record {
    identity: RemoteIdentity,
    /// Held ranges, `[start, end)`, sorted and apart.
    ranges: Vec<(u64, u64)>,
}

pub struct SparseCopy {
    path: PathBuf,
    record_path: PathBuf,
    identity: RemoteIdentity,
    fetcher: Box<dyn RangeFetch>,
    file: File,
    /// Held shared while the copy is open.
    _lock: File,
    /// Held ranges by start.
    held: Mutex<BTreeMap<u64, u64>>,
}

impl SparseCopy {
    /// Opens the copy at `path`, or starts one, of the remote file
    /// `identity` names.
    pub fn open(
        path: &Path,
        identity: RemoteIdentity,
        fetcher: Box<dyn RangeFetch>,
    ) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let record_path = sidecar(path, "ranges");
        let lock_path = sidecar(path, "lock");
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("open {}", lock_path.display()))?;
        // twice: an eviction may remove the copy between the two locks
        let mut opened = None;
        for _ in 0..2 {
            start_over_unless_open(&lock, &lock_path, path, &record_path, &identity)?;
            lock.lock_shared()
                .with_context(|| format!("lock {}", lock_path.display()))?;
            // read under the shared lock: no opener starts the copy over now
            opened = open_held(path, &record_path, &identity);
            if opened.is_some() {
                break;
            }
            lock.unlock()
                .with_context(|| format!("unlock {}", lock_path.display()))?;
        }
        let Some((file, held)) = opened else {
            bail!(
                "the sparse copy {} is open for another file, or is damaged while in use",
                path.display()
            );
        };
        // the last use, which eviction removes the oldest copies by
        let _ = lock.set_modified(std::time::SystemTime::now());
        Ok(Self {
            path: path.to_owned(),
            record_path,
            identity,
            fetcher,
            file,
            _lock: lock,
            held: Mutex::new(held),
        })
    }

    /// The copy's path, for a loader that reads the GGUF itself.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes of the remote file the copy holds.
    pub fn held_bytes(&self) -> u64 {
        self.held
            .lock()
            .map(|held| held.iter().map(|(start, end)| end - start).sum())
            .unwrap_or(0)
    }

    /// Fetches what the copy lacks of `ranges`, `(offset, len)` each, many
    /// requests at once.
    pub fn fetch(&self, ranges: &[(u64, u64)]) -> Result<()> {
        let plan = {
            let held = self.lock()?;
            fetch_plan(&held, ranges, self.identity.size)?
        };
        if plan.is_empty() {
            return Ok(());
        }
        let fetchers = FETCHERS.min(plan.len());
        let queue = Mutex::new(plan.into_iter());
        let fetched = Mutex::new(Vec::new());
        let failure = Mutex::new(None);
        if fetchers == 1 {
            // one request needs no thread
            self.fetch_from(&queue, &fetched, &failure);
        } else {
            std::thread::scope(|scope| {
                for _ in 0..fetchers {
                    scope.spawn(|| self.fetch_from(&queue, &fetched, &failure));
                }
            });
        }
        // what did fetch stays useful, and is recorded, when a range failed
        let saved = self.save(fetched.into_inner().unwrap_or_default());
        match failure.into_inner().ok().flatten() {
            Some(error) => Err(error),
            None => saved,
        }
    }

    fn fetch_from(
        &self,
        queue: &Mutex<std::vec::IntoIter<(u64, u64)>>,
        fetched: &Mutex<Vec<(u64, u64)>>,
        failure: &Mutex<Option<anyhow::Error>>,
    ) {
        loop {
            let next = queue.lock().ok().and_then(|mut queue| queue.next());
            let Some((start, end)) = next else {
                return;
            };
            match self.fetch_one(start, end) {
                Ok(()) => {
                    if let Ok(mut fetched) = fetched.lock() {
                        fetched.push((start, end));
                    }
                }
                Err(error) => {
                    if let Ok(mut failure) = failure.lock() {
                        failure.get_or_insert(error);
                    }
                    // the other ranges still fetch: they stay useful
                }
            }
        }
    }

    fn fetch_one(&self, start: u64, end: u64) -> Result<()> {
        let bytes = self
            .fetcher
            .fetch(start, end - start)
            .with_context(|| format!("fetch bytes {start}..{end} of {}", self.identity.source))?;
        if bytes.len() as u64 != end - start {
            bail!(
                "{} sent {} bytes for {start}..{end}",
                self.identity.source,
                bytes.len()
            );
        }
        write_at(&self.file, &bytes, start)
    }

    /// Records `fetched` ranges as held, once their bytes are on disk.
    fn save(&self, fetched: Vec<(u64, u64)>) -> Result<()> {
        if fetched.is_empty() {
            return Ok(());
        }
        // the bytes first: a record of bytes a crash loses would have the
        // copy read zeros where the file has weights
        self.file
            .sync_data()
            .with_context(|| format!("sync {}", self.path.display()))?;
        let record = {
            let mut held = self.lock()?;
            for (start, end) in fetched {
                insert(&mut held, start, end);
            }
            // another opener's record names bytes it synced to this file
            if let Some(theirs) = read_held(&self.record_path, &self.identity) {
                for (start, end) in theirs {
                    insert(&mut held, start, end);
                }
            }
            Record {
                identity: self.identity.clone(),
                ranges: held.iter().map(|(&s, &e)| (s, e)).collect(),
            }
        };
        write_record(&self.record_path, &record)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, BTreeMap<u64, u64>>> {
        self.held
            .lock()
            .map_err(|_| anyhow::anyhow!("sparse copy lock poisoned"))
    }

    /// `[offset, offset + len)` as a range inside the file.
    fn range(&self, offset: u64, len: u64) -> Result<(u64, u64)> {
        match offset.checked_add(len) {
            Some(end) if end <= self.identity.size => Ok((offset, end)),
            _ => bail!(
                "bytes {offset}+{len} are past the end of the file ({} bytes)",
                self.identity.size
            ),
        }
    }
}

impl ByteSource for SparseCopy {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.range(offset, len)?;
        self.fetch(&[(offset, len)])?;
        let mut buf = vec![0u8; usize::try_from(len)?];
        read_at(&self.file, &mut buf, offset)?;
        Ok(buf)
    }

    fn size(&self) -> u64 {
        self.identity.size
    }

    fn prefetch(&self, ranges: &[(u64, u64)]) -> Result<()> {
        self.fetch(ranges)
    }

    fn read_once(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let (start, end) = self.range(offset, len)?;
        let held = lacking(&*self.lock()?, start, end).is_empty();
        if held {
            let mut buf = vec![0u8; usize::try_from(len)?];
            read_at(&self.file, &mut buf, offset)?;
            return Ok(buf);
        }
        let bytes = self.fetcher.fetch(offset, len)?;
        if bytes.len() as u64 != len {
            bail!(
                "{} sent {} bytes, not {len}",
                self.identity.source,
                bytes.len()
            );
        }
        Ok(bytes)
    }

    fn missing_bytes(&self, ranges: &[(u64, u64)]) -> Result<u64> {
        let held = self.lock()?;
        let missing = missing(&held, ranges, self.identity.size)?;
        Ok(missing.iter().map(|(start, end)| end - start).sum())
    }
}

/// `path` with `.suffix` added to its file name.
fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(".");
    name.push(suffix);
    path.with_file_name(name)
}

/// Opens the data file of a copy of `identity` and the ranges its record
/// says it holds. `None` when the copy is not one of `identity`: no record,
/// a record of another file or one that does not read, or no data file of
/// the file's size.
fn open_held(
    path: &Path,
    record_path: &Path,
    identity: &RemoteIdentity,
) -> Option<(File, BTreeMap<u64, u64>)> {
    let held = read_held(record_path, identity)?;
    let file = OpenOptions::new().read(true).write(true).open(path).ok()?;
    (file.metadata().ok()?.len() == identity.size).then_some((file, held))
}

/// The ranges the record at `path` says a copy of `identity` holds.
fn read_held(path: &Path, identity: &RemoteIdentity) -> Option<BTreeMap<u64, u64>> {
    let record: Record = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    if record.identity != *identity {
        return None;
    }
    let mut held = BTreeMap::new();
    for (start, end) in record.ranges {
        if start >= end || end > identity.size {
            return None;
        }
        insert(&mut held, start, end);
    }
    Some(held)
}

/// Starts the copy at `path` over when it is not one of `identity` and no
/// other opener holds it: under the exclusive lock of `lock`.
fn start_over_unless_open(
    lock: &File,
    lock_path: &Path,
    path: &Path,
    record_path: &Path,
    identity: &RemoteIdentity,
) -> Result<()> {
    match lock.try_lock() {
        Ok(()) => {
            if open_held(path, record_path, identity).is_none() {
                start_over(path, record_path, identity)?;
            }
            lock.unlock()
                .with_context(|| format!("unlock {}", lock_path.display()))
        }
        Err(TryLockError::WouldBlock) => Ok(()),
        Err(TryLockError::Error(error)) => {
            Err(error).with_context(|| format!("lock {}", lock_path.display()))
        }
    }
}

/// Makes the copy at `path` an empty copy of `identity`. Only under the
/// exclusive lock: no opener has the data file open.
fn start_over(path: &Path, record_path: &Path, identity: &RemoteIdentity) -> Result<()> {
    // the record first: none may name bytes that are about to go
    match std::fs::remove_file(record_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("remove {}", record_path.display()));
        }
    }
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    file.set_len(identity.size)
        .with_context(|| format!("size {}", path.display()))?;
    let record = Record {
        identity: identity.clone(),
        ranges: Vec::new(),
    };
    write_record(record_path, &record)
}

/// Writes `record` to `path` through a temporary file of its own, so two
/// writers never share one, and renames it into place.
fn write_record(path: &Path, record: &Record) -> Result<()> {
    /// Tells apart the temporary files of one process.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let temp = sidecar(
        path,
        &format!(
            "{}.{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ),
    );
    // a file a dead process of the same id left: it is no one's now
    let _ = std::fs::remove_file(&temp);
    let written = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(&serde_json::to_vec(record)?)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&temp);
        return Err(error).with_context(|| format!("write {}", path.display()));
    }
    // the new name too, where the platform can sync a directory: without
    // it a crash may bring the older record back, which names fewer ranges
    if let Some(parent) = path.parent()
        && let Ok(dir) = File::open(parent)
    {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// What `held` lacks of `wanted`, `(offset, len)` each: ranges
/// `[start, end)`, sorted and apart.
fn missing(held: &BTreeMap<u64, u64>, wanted: &[(u64, u64)], size: u64) -> Result<Vec<(u64, u64)>> {
    let mut ranges = Vec::with_capacity(wanted.len());
    for &(offset, len) in wanted {
        let end = offset.checked_add(len).filter(|&end| end <= size);
        let Some(end) = end else {
            bail!("bytes {offset}+{len} are past the end of the file ({size} bytes)");
        };
        if len > 0 {
            ranges.push((offset, end));
        }
    }
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    let mut covered = 0;
    for (start, end) in ranges {
        // what an earlier range covered is not missing twice
        let start = start.max(covered);
        if start < end {
            out.extend(lacking(held, start, end));
            covered = end;
        }
    }
    Ok(out)
}

/// The ranges to fetch for `wanted`: what `held` lacks, joined across small
/// gaps and split into requests of at most [`MAX_FETCH`].
fn fetch_plan(
    held: &BTreeMap<u64, u64>,
    wanted: &[(u64, u64)],
    size: u64,
) -> Result<Vec<(u64, u64)>> {
    let mut joined: Vec<(u64, u64)> = Vec::new();
    for (start, end) in missing(held, wanted, size)? {
        match joined.last_mut() {
            Some(last) if start <= last.1.saturating_add(MERGE_GAP) => last.1 = last.1.max(end),
            _ => joined.push((start, end)),
        }
    }
    Ok(joined
        .into_iter()
        .flat_map(|(start, end)| {
            (start..end)
                .step_by(MAX_FETCH as usize)
                .map(move |s| (s, s.saturating_add(MAX_FETCH).min(end)))
        })
        .collect())
}

/// The parts of `[start, end)` that `held` lacks.
fn lacking(held: &BTreeMap<u64, u64>, start: u64, end: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    if start >= end {
        // nothing is wanted, and a range of the map may not end before it starts
        return out;
    }
    let mut at = start;
    let before = held.range(..=start).next_back().map(|(&s, &e)| (s, e));
    for (s, e) in before
        .into_iter()
        .chain(held.range(start + 1..end).map(|(&s, &e)| (s, e)))
    {
        if e <= at {
            continue;
        }
        if s > at {
            out.push((at, s.min(end)));
        }
        at = at.max(e);
        if at >= end {
            return out;
        }
    }
    if at < end {
        out.push((at, end));
    }
    out
}

/// Adds `[start, end)` to `held`, joining the ranges it meets.
fn insert(held: &mut BTreeMap<u64, u64>, mut start: u64, mut end: u64) {
    if let Some((&s, &e)) = held.range(..=start).next_back()
        && e >= start
    {
        start = s;
        end = end.max(e);
    }
    let met: Vec<u64> = held.range(start..=end).map(|(&s, _)| s).collect();
    for s in met {
        if let Some(e) = held.remove(&s) {
            end = end.max(e);
        }
    }
    held.insert(start, end);
}

#[cfg(unix)]
fn write_at(file: &File, bytes: &[u8], offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    Ok(file.write_all_at(bytes, offset)?)
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    Ok(file.read_exact_at(buf, offset)?)
}

#[cfg(windows)]
fn write_at(file: &File, mut bytes: &[u8], mut offset: u64) -> Result<()> {
    use std::os::windows::fs::FileExt;
    while !bytes.is_empty() {
        let n = file.seek_write(bytes, offset)?;
        if n == 0 {
            bail!("the sparse copy took no more bytes at {offset}");
        }
        bytes = &bytes[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn read_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_read(buf, offset)?;
        if n == 0 {
            bail!("read past the end of the sparse copy");
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
