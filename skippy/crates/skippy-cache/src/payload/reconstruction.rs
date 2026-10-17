//! Assemble fragmented snapshots without serializing large host-memory copies.
use std::thread;

use anyhow::{Context, Result, ensure};

use super::bytes::{CacheBlockBytes, CacheBlockRef};

const PARALLEL_COPY_MIN_BYTES: usize = 64 * 1024 * 1024;
const MAX_COPY_WORKERS: usize = 8;

pub(super) fn reconstruct_blocks(blocks: &[CacheBlockRef], len: usize) -> Result<Vec<u8>> {
    let workers = if len >= PARALLEL_COPY_MIN_BYTES {
        thread::available_parallelism().map_or(1, usize::from)
    } else {
        1
    };
    reconstruct_with_workers(blocks, len, workers)
}

fn snapshot_blocks(blocks: &[CacheBlockRef], len: usize) -> Result<Vec<CacheBlockBytes>> {
    let mut total = 0usize;
    let mut snapshots = Vec::with_capacity(blocks.len());
    for block in blocks {
        let bytes = block
            .bytes
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let slice = bytes
            .storage
            .get(bytes.range.clone())
            .context("cache block range exceeds its backing storage")?;
        total = total
            .checked_add(slice.len())
            .context("cache payload too large to reconstruct")?;
        // Pin immutable storage, releasing each metadata lock before copying.
        // Eviction may relocate a block while these readers retain its bytes.
        snapshots.push(bytes.clone());
    }
    ensure!(
        total == len,
        "cache payload reconstruction length mismatch: expected {len} got {total}"
    );
    Ok(snapshots)
}

fn copy_blocks(blocks: &[CacheBlockBytes], mut output: &mut [u8]) {
    for block in blocks {
        let source = block.as_slice();
        let (destination, tail) = output.split_at_mut(source.len());
        destination.copy_from_slice(source);
        output = tail;
    }
    debug_assert!(output.is_empty());
}

fn copy_parallel(
    blocks: &[CacheBlockBytes],
    mut output: &mut [u8],
    workers: usize,
) -> std::io::Result<()> {
    thread::scope(move |scope| {
        for group in blocks.chunks(blocks.len().div_ceil(workers)) {
            let bytes = group.iter().map(|block| block.range.len()).sum();
            let (destination, tail) = output.split_at_mut(bytes);
            output = tail;
            thread::Builder::new()
                .name("cache-reconstruct".into())
                .spawn_scoped(scope, move || copy_blocks(group, destination))?;
        }
        Ok(())
    })
}

fn reconstruct_with_workers(
    blocks: &[CacheBlockRef],
    len: usize,
    workers: usize,
) -> Result<Vec<u8>> {
    let snapshots = snapshot_blocks(blocks, len)?;
    // Large zeroed allocations retain lazy page initialization; workers touch
    // disjoint destinations, without unsafe uninitialized-memory handling.
    let mut output = vec![0; len];
    let workers = workers.clamp(1, MAX_COPY_WORKERS).min(snapshots.len());
    if workers <= 1 || copy_parallel(&snapshots, &mut output, workers).is_err() {
        // A thread-creation failure is transient, not payload corruption. The
        // scope joins started workers before retrying the complete copy here.
        copy_blocks(&snapshots, &mut output);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;

    #[test]
    fn parallel_reconstruction_preserves_unequal_ranges_and_shared_storage() {
        let storage = Arc::new((0..8192).map(|i| (i % 251) as u8).collect::<Vec<_>>());
        let ranges = [7..19, 4096..8001, 3..4, 31..88, 0..4096];
        let expected: Vec<_> = ranges
            .iter()
            .flat_map(|range| &storage[range.clone()])
            .copied()
            .collect();
        let blocks: Vec<_> = ranges
            .into_iter()
            .enumerate()
            .map(|(i, range)| {
                CacheBlockRef::new(
                    i.to_string(),
                    Arc::new(RwLock::new(CacheBlockBytes::new(
                        Arc::clone(&storage),
                        range,
                    ))),
                )
            })
            .collect();
        for workers in [1, 2, 8, usize::MAX] {
            assert_eq!(
                reconstruct_with_workers(&blocks, expected.len(), workers).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn reconstruction_rejects_invalid_lengths_before_copying() {
        let block = CacheBlockRef::new(
            "block".into(),
            Arc::new(RwLock::new(CacheBlockBytes::new(
                Arc::new(vec![1, 2, 3]),
                0..3,
            ))),
        );
        for len in [0, 2, 4, usize::MAX] {
            assert!(reconstruct_with_workers(std::slice::from_ref(&block), len, 8).is_err());
        }
        assert!(reconstruct_with_workers(&[], 0, 8).unwrap().is_empty());
    }

    #[test]
    fn reconstruction_rejects_invalid_backing_ranges() {
        let block = CacheBlockRef::new(
            "block".into(),
            Arc::new(RwLock::new(CacheBlockBytes {
                storage: Arc::new(vec![1, 2, 3]),
                range: 0..4,
            })),
        );
        assert!(reconstruct_with_workers(&[block], 4, 8).is_err());
    }
}
