//! Random-read throughput of the local model store.
//!
//! Split planning can keep a mixture-of-experts model on fewer nodes by
//! letting routed experts page in from disk per token. Whether that beats a
//! pipeline hop depends on how fast the drive holding the model serves random
//! reads of expert-sized chunks, so the node measures it once at startup and
//! gossips the result with its memory block. The probe reads an existing
//! large file in the store rather than writing anything.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// One expert bank slice is tens of megabytes; read in that shape.
const CHUNK_BYTES: u64 = 16 * 1024 * 1024;
/// Files smaller than this do not spread reads across enough of the device.
const MIN_PROBE_FILE_BYTES: u64 = 16 * CHUNK_BYTES;
/// Stop after this many chunks or this much wall time, whichever comes first.
const MAX_CHUNKS: u32 = 64;
const MAX_PROBE_DURATION: Duration = Duration::from_millis(1500);
/// Parallel readers, so NVMe queues see more than one outstanding request.
const READERS: usize = 4;

/// Measured random-read throughput, in bytes per second.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageReadRate {
    pub bytes_per_second: u64,
    pub bytes_read: u64,
    pub elapsed: Duration,
}

/// Measure random 16 MiB reads against the largest file under `store`.
///
/// Returns `None` when the store has no file large enough to probe. The
/// result is a lower bound on sequential throughput and an upper bound on
/// what per-token expert paging will see, which is the right side to err on
/// for a placement decision.
pub fn measure_model_store_read_rate(store: &Path) -> Option<StorageReadRate> {
    let target = largest_file_under(store)?;
    measure_file_read_rate(&target)
}

/// Measure random reads against one specific file.
pub fn measure_file_read_rate(path: &Path) -> Option<StorageReadRate> {
    let len = std::fs::metadata(path).ok()?.len();
    if len < MIN_PROBE_FILE_BYTES {
        return None;
    }
    let chunk_slots = len / CHUNK_BYTES;
    let chunks_per_reader = (u64::from(MAX_CHUNKS) / READERS as u64).max(1);
    let started = Instant::now();
    let results = std::thread::scope(|scope| {
        let workers = (0..READERS)
            .map(|reader| {
                let path: PathBuf = path.to_path_buf();
                scope.spawn(move || -> Option<u64> {
                    let file = open_uncached(&path).ok()?;
                    let mut buffer = vec![0u8; CHUNK_BYTES as usize];
                    let mut read = 0u64;
                    let mut sequence = stride_sequence(reader as u64, chunk_slots);
                    for _ in 0..chunks_per_reader {
                        if started.elapsed() > MAX_PROBE_DURATION {
                            break;
                        }
                        let slot = sequence.next()?;
                        read += read_chunk_at(&file, slot * CHUNK_BYTES, &mut buffer)?;
                    }
                    Some(read)
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().ok().flatten())
            .collect::<Vec<_>>()
    });
    let elapsed = started.elapsed();
    let bytes_read = results.into_iter().flatten().sum::<u64>();
    if bytes_read == 0 || elapsed.is_zero() {
        return None;
    }
    let bytes_per_second = (u128::from(bytes_read) * 1_000_000_000 / elapsed.as_nanos().max(1))
        .min(u128::from(u64::MAX)) as u64;
    Some(StorageReadRate {
        bytes_per_second,
        bytes_read,
        elapsed,
    })
}

/// Visit chunk slots in a fixed-stride order that spreads each reader across
/// the file without a random-number dependency.
fn stride_sequence(reader: u64, chunk_slots: u64) -> impl Iterator<Item = u64> {
    // A large odd stride walks every slot of the ring before repeating.
    const STRIDE: u64 = 7_919;
    let slots = chunk_slots.max(1);
    (0u64..).map(move |step| (reader + step * STRIDE) % slots)
}

fn largest_file_under(root: &Path) -> Option<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() {
                let len = metadata.len();
                if best.as_ref().is_none_or(|(size, _)| len > *size) {
                    best = Some((len, path));
                }
            }
        }
    }
    best.filter(|(len, _)| *len >= MIN_PROBE_FILE_BYTES)
        .map(|(_, path)| path)
}

#[cfg(target_os = "macos")]
fn open_uncached(path: &Path) -> std::io::Result<File> {
    use std::os::fd::AsRawFd;
    let file = File::open(path)?;
    // SAFETY: fcntl on a file descriptor this process owns; F_NOCACHE only
    // changes caching policy and cannot invalidate memory.
    unsafe {
        libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
    }
    Ok(file)
}

#[cfg(target_os = "linux")]
fn open_uncached(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    // O_DIRECT needs aligned buffers; fall back to a cached open when the
    // filesystem refuses it rather than failing the probe.
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .or_else(|_| File::open(path))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn open_uncached(path: &Path) -> std::io::Result<File> {
    File::open(path)
}

#[cfg(unix)]
fn read_chunk_at(file: &File, offset: u64, buffer: &mut [u8]) -> Option<u64> {
    use std::os::unix::fs::FileExt;
    file.read_at(buffer, offset).ok().map(|read| read as u64)
}

#[cfg(windows)]
fn read_chunk_at(file: &File, offset: u64, buffer: &mut [u8]) -> Option<u64> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buffer, offset).ok().map(|read| read as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_sequence_covers_every_slot_before_repeating() {
        let slots = 13;
        let mut seen = std::collections::BTreeSet::new();
        for slot in stride_sequence(3, slots).take(slots as usize) {
            seen.insert(slot);
        }
        assert_eq!(seen.len(), slots as usize);
    }

    #[test]
    fn small_files_are_not_probed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.bin");
        std::fs::write(&path, vec![0u8; 1024]).unwrap();
        assert_eq!(measure_file_read_rate(&path), None);
        assert_eq!(measure_model_store_read_rate(dir.path()), None);
    }

    #[test]
    fn files_below_the_probe_floor_are_never_targets() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("nested")).unwrap();
        std::fs::write(dir.path().join("a.bin"), vec![0u8; 10]).unwrap();
        std::fs::write(dir.path().join("nested/b.bin"), vec![0u8; 20]).unwrap();
        assert_eq!(largest_file_under(dir.path()), None);
    }

    #[test]
    fn the_largest_file_in_a_nested_store_is_measured() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("nested")).unwrap();
        let small = dir.path().join("small.bin");
        let large = dir.path().join("nested/large.bin");
        std::fs::File::create(&small)
            .unwrap()
            .set_len(MIN_PROBE_FILE_BYTES)
            .unwrap();
        std::fs::File::create(&large)
            .unwrap()
            .set_len(MIN_PROBE_FILE_BYTES * 2)
            .unwrap();
        assert_eq!(largest_file_under(dir.path()), Some(large.clone()));

        let rate = measure_model_store_read_rate(dir.path()).expect("probe a sparse file");
        assert!(rate.bytes_read > 0);
        assert!(rate.bytes_per_second > 0);
    }
}
