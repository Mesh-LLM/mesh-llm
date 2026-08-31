use std::{collections::HashMap, sync::Arc, time::Instant};

use anyhow::{Result, bail};

use super::CacheBytes;
use super::bytes::{CacheBlockRef, CacheBytesRepr};

const DEFAULT_BLOCK_SIZE_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub struct CacheBlobStore {
    block_size: usize,
    physical_bytes: u64,
    blocks: HashMap<String, CacheBlob>,
}

impl Default for CacheBlobStore {
    fn default() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE_BYTES)
    }
}

#[derive(Debug)]
struct CacheBlob {
    bytes: Arc<Vec<u8>>,
    ref_count: u64,
}

impl CacheBlobStore {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            physical_bytes: 0,
            blocks: HashMap::new(),
        }
    }

    pub fn store_bytes(&mut self, bytes: CacheBytes) -> (CacheBytes, CacheDedupeStats) {
        let len = bytes.len;
        let bytes = match bytes.repr {
            CacheBytesRepr::Inline(bytes) => bytes,
            CacheBytesRepr::Blocks(blocks) => {
                let mut stats = CacheDedupeStats::default();
                for block in blocks.iter() {
                    stats.block_count = stats.block_count.saturating_add(1);
                    let entry = self.blocks.entry(block.hash.clone()).or_insert_with(|| {
                        self.physical_bytes =
                            self.physical_bytes.saturating_add(block.bytes.len() as u64);
                        stats.new_block_count = stats.new_block_count.saturating_add(1);
                        CacheBlob {
                            bytes: block.bytes.clone(),
                            ref_count: 0,
                        }
                    });
                    if entry.ref_count > 0 {
                        stats.reused_block_count = stats.reused_block_count.saturating_add(1);
                    }
                    entry.ref_count = entry.ref_count.saturating_add(1);
                }
                return (
                    CacheBytes {
                        len,
                        repr: CacheBytesRepr::Blocks(blocks),
                    },
                    stats,
                );
            }
        };
        let mut blocks = Vec::new();
        let started = Instant::now();
        let mut stats = CacheDedupeStats {
            hash_bytes: bytes.len() as u64,
            ..CacheDedupeStats::default()
        };
        for chunk in bytes.chunks(self.block_size) {
            stats.block_count = stats.block_count.saturating_add(1);
            let hash = blake3::hash(chunk).to_hex().to_string();
            let entry = self.blocks.entry(hash.clone()).or_insert_with(|| {
                self.physical_bytes = self.physical_bytes.saturating_add(chunk.len() as u64);
                stats.new_block_count = stats.new_block_count.saturating_add(1);
                CacheBlob {
                    bytes: Arc::new(chunk.to_vec()),
                    ref_count: 0,
                }
            });
            if entry.ref_count > 0 {
                stats.reused_block_count = stats.reused_block_count.saturating_add(1);
            }
            entry.ref_count = entry.ref_count.saturating_add(1);
            blocks.push(CacheBlockRef::new(hash, entry.bytes.clone()));
        }
        stats.hash_ms = started.elapsed().as_secs_f64() * 1000.0;
        (CacheBytes::blocks(bytes.len() as u64, blocks), stats)
    }

    pub fn release_bytes(&mut self, bytes: &CacheBytes) -> Result<()> {
        self.release_bytes_batch(&[bytes])
    }

    pub(crate) fn release_bytes_batch(&mut self, payloads: &[&CacheBytes]) -> Result<()> {
        let mut releases = HashMap::<String, u64>::new();
        for bytes in payloads {
            for hash in bytes.block_hashes() {
                let count = releases.entry(hash.to_string()).or_default();
                *count = count
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("cache blob release count overflow"))?;
            }
        }
        let mut physical_bytes_to_remove = 0u64;
        for (hash, count) in &releases {
            let Some(entry) = self.blocks.get(hash) else {
                bail!("cache blob release references missing block {hash}");
            };
            if entry.ref_count < *count {
                bail!(
                    "cache blob release underflow for block {hash}: refs={} releases={count}",
                    entry.ref_count
                );
            }
            if entry.ref_count == *count {
                physical_bytes_to_remove = physical_bytes_to_remove
                    .checked_add(entry.bytes.len() as u64)
                    .ok_or_else(|| anyhow::anyhow!("cache blob physical byte count overflow"))?;
            }
        }
        if physical_bytes_to_remove > self.physical_bytes {
            bail!(
                "cache blob physical byte accounting underflow: bytes={} releases={physical_bytes_to_remove}",
                self.physical_bytes
            );
        }
        for (hash, count) in releases {
            let entry = self
                .blocks
                .get_mut(&hash)
                .expect("release prevalidation guarantees block presence");
            entry.ref_count -= count;
            if entry.ref_count == 0 {
                self.blocks.remove(&hash);
            }
        }
        self.physical_bytes -= physical_bytes_to_remove;
        Ok(())
    }

    pub fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn logical_ref_count(&self) -> u64 {
        self.blocks
            .values()
            .map(|block| block.ref_count)
            .fold(0, u64::saturating_add)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheDedupeStats {
    pub hash_ms: f64,
    pub hash_bytes: u64,
    pub block_count: usize,
    pub new_block_count: usize,
    pub reused_block_count: usize,
}

impl CacheDedupeStats {
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            hash_ms: self.hash_ms + other.hash_ms,
            hash_bytes: self.hash_bytes.saturating_add(other.hash_bytes),
            block_count: self.block_count.saturating_add(other.block_count),
            new_block_count: self.new_block_count.saturating_add(other.new_block_count),
            reused_block_count: self
                .reused_block_count
                .saturating_add(other.reused_block_count),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::payload::{CacheBlobStore, ExactStatePayload};

    #[test]
    fn block_store_dedupes_repeated_payload_blocks() {
        let mut blobs = CacheBlobStore::new(4);
        let first = ExactStatePayload::full_state(b"aaaabbbb".to_vec());
        let second = ExactStatePayload::full_state(b"aaaacccc".to_vec());

        let (_, first_stats) = first.dedupe_into(&mut blobs);
        let (second, second_stats) = second.dedupe_into(&mut blobs);

        assert_eq!(first_stats.new_block_count, 2);
        assert_eq!(second_stats.new_block_count, 1);
        assert_eq!(second_stats.reused_block_count, 1);
        assert_eq!(blobs.physical_bytes(), 12);
        assert_eq!(second.byte_len(), 8);
    }

    #[test]
    fn rededuping_block_payload_retains_a_logical_owner() {
        let mut blobs = CacheBlobStore::new(4);
        let original = ExactStatePayload::full_state(b"aaaabbbb".to_vec());
        let (first_owner, _) = original.dedupe_into(&mut blobs);
        let (second_owner, stats) = first_owner.clone().dedupe_into(&mut blobs);

        assert_eq!(stats.reused_block_count, 2);
        assert_eq!(blobs.physical_bytes(), 8);
        first_owner.release_from(&mut blobs).unwrap();
        assert_eq!(blobs.physical_bytes(), 8);
        assert_eq!(
            second_owner.full_state_bytes_timed().unwrap().0.as_ref(),
            b"aaaabbbb"
        );
        second_owner.release_from(&mut blobs).unwrap();
        assert_eq!(blobs.physical_bytes(), 0);
    }

    #[test]
    fn duplicate_release_is_reported_without_accounting_drift() {
        let mut blobs = CacheBlobStore::new(4);
        let (payload, _) =
            ExactStatePayload::full_state(b"aaaabbbb".to_vec()).dedupe_into(&mut blobs);

        payload.release_from(&mut blobs).unwrap();
        let error = payload.release_from(&mut blobs).unwrap_err();

        assert!(error.to_string().contains("missing block"));
        assert_eq!(blobs.physical_bytes(), 0);
        assert_eq!(blobs.block_count(), 0);
    }

    #[test]
    fn composite_release_prevalidates_every_component() {
        let mut blobs = CacheBlobStore::new(4);
        let (payload, _) =
            ExactStatePayload::kv_recurrent(b"aaaabbbb".to_vec(), b"ccccdddd".to_vec())
                .dedupe_into(&mut blobs);
        let ExactStatePayload::KvRecurrent { recurrent, .. } = &payload else {
            panic!("expected composite payload");
        };

        blobs.release_bytes(recurrent).unwrap();
        let physical_before = blobs.physical_bytes();
        let refs_before = blobs.logical_ref_count();

        let error = payload.release_from(&mut blobs).unwrap_err();

        assert!(error.to_string().contains("missing block"));
        assert_eq!(blobs.physical_bytes(), physical_before);
        assert_eq!(blobs.logical_ref_count(), refs_before);
        assert_eq!(blobs.block_count(), 2);
    }

    #[test]
    fn physical_accounting_underflow_does_not_mutate_references() {
        let mut blobs = CacheBlobStore::new(4);
        let (payload, _) =
            ExactStatePayload::full_state(b"aaaabbbb".to_vec()).dedupe_into(&mut blobs);
        let refs_before = blobs.logical_ref_count();
        blobs.physical_bytes = 0;

        let error = payload.release_from(&mut blobs).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("physical byte accounting underflow")
        );
        assert_eq!(blobs.logical_ref_count(), refs_before);
        assert_eq!(blobs.block_count(), 2);
    }
}
