//! A local copy of a remote GGUF that holds only the bytes an audit read.
//!
//! The copy is a file of the remote file's size. The ranges it holds are
//! fetched on first use and recorded next to it, and the rest stays a hole,
//! so it takes no disk space. The glue check loads a layer from the copy as
//! from a whole GGUF: it never reads the weight matrices, whose opened
//! outputs replace them, and the holes read as zeros.
//!
//! The record names the remote file by its pinned revision and size. A
//! copy whose record names another file starts over. Two processes that
//! fill the same copy write the same bytes; one may lose the other's record
//! of a range and fetch it again.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::weights::ByteSource;

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
        let record_path = record_path(path);
        let held = match read_record(&record_path) {
            Some(record) if record.identity == identity => record.ranges.into_iter().collect(),
            _ => BTreeMap::new(),
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;
        if held.is_empty() {
            // a fresh copy: no bytes of another file may stay
            file.set_len(0)?;
        }
        file.set_len(identity.size)?;
        Ok(Self {
            path: path.to_owned(),
            record_path,
            identity,
            fetcher,
            file,
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
        let missing = {
            let held = self.lock()?;
            fetch_plan(&held, ranges, self.identity.size)?
        };
        if missing.is_empty() {
            return Ok(());
        }
        let queue = Mutex::new(missing.into_iter());
        let fetched = Mutex::new(Vec::new());
        let failure = Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..FETCHERS {
                scope.spawn(|| self.fetch_from(&queue, &fetched, &failure));
            }
        });
        let fetched = fetched.into_inner().unwrap_or_default();
        {
            let mut held = self.lock()?;
            for (start, end) in fetched {
                insert(&mut held, start, end);
            }
        }
        self.save()?;
        match failure.into_inner().ok().flatten() {
            Some(error) => Err(error),
            None => Ok(()),
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

    fn save(&self) -> Result<()> {
        let record = Record {
            identity: self.identity.clone(),
            ranges: self.lock()?.iter().map(|(&s, &e)| (s, e)).collect(),
        };
        let temp = self
            .record_path
            .with_extension(format!("ranges.{}.tmp", std::process::id()));
        std::fs::write(&temp, serde_json::to_vec(&record)?)
            .with_context(|| format!("write {}", temp.display()))?;
        std::fs::rename(&temp, &self.record_path)
            .with_context(|| format!("write {}", self.record_path.display()))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, BTreeMap<u64, u64>>> {
        self.held
            .lock()
            .map_err(|_| anyhow::anyhow!("sparse copy lock poisoned"))
    }
}

impl ByteSource for SparseCopy {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
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
        let held = lacking(&*self.lock()?, offset, offset.saturating_add(len)).is_empty();
        if held {
            let mut buf = vec![0u8; usize::try_from(len)?];
            read_at(&self.file, &mut buf, offset)?;
            return Ok(buf);
        }
        if offset.saturating_add(len) > self.identity.size {
            bail!(
                "bytes {offset}..{} are past the end of the file",
                offset + len
            );
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
}

fn record_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(".ranges");
    path.with_file_name(name)
}

fn read_record(path: &Path) -> Option<Record> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// The ranges to fetch for `wanted`: what `held` lacks, joined across small
/// gaps and split into requests of at most [`MAX_FETCH`].
fn fetch_plan(
    held: &BTreeMap<u64, u64>,
    wanted: &[(u64, u64)],
    size: u64,
) -> Result<Vec<(u64, u64)>> {
    let mut missing: Vec<(u64, u64)> = Vec::new();
    let mut wanted: Vec<(u64, u64)> = wanted
        .iter()
        .filter(|&&(_, len)| len > 0)
        .map(|&(offset, len)| (offset, offset.saturating_add(len)))
        .collect();
    wanted.sort_unstable();
    for (start, end) in wanted {
        if end > size {
            bail!("bytes {start}..{end} are past the end of the file ({size} bytes)");
        }
        for (start, end) in lacking(held, start, end) {
            match missing.last_mut() {
                Some(last) if start <= last.1 + MERGE_GAP => last.1 = last.1.max(end),
                _ => missing.push((start, end)),
            }
        }
    }
    Ok(missing
        .into_iter()
        .flat_map(|(start, end)| {
            (start..end)
                .step_by(MAX_FETCH as usize)
                .map(move |s| (s, (s + MAX_FETCH).min(end)))
        })
        .collect())
}

/// The parts of `[start, end)` that `held` lacks.
fn lacking(held: &BTreeMap<u64, u64>, start: u64, end: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
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
