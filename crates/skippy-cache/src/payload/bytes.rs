use std::{borrow::Cow, sync::Arc, time::Instant};

use anyhow::{Result, anyhow};
use memmap2::Mmap;

#[derive(Debug, Clone)]
pub struct CacheBytes {
    pub(super) len: u64,
    pub(super) repr: CacheBytesRepr,
}

#[derive(Debug, Clone)]
pub(super) enum CacheBytesRepr {
    Inline(Arc<Vec<u8>>),
    Blocks(Arc<[CacheBlockRef]>),
    /// A byte range inside a memory-mapped file on the disk tier.
    ///
    /// This exists so restoring a large prefix is a *borrow*, not a copy.
    /// `skippy_import_kv_page` takes a contiguous `(ptr, len)`, so a mapped
    /// range can be handed straight to the runtime and the kernel faults in
    /// only the pages actually touched. A page still warm in the page cache
    /// costs almost nothing; a cold one costs a single sequential read.
    ///
    /// Deliberately *not* block-deduplicated. `as_cow()` on a `Blocks` payload
    /// allocates and concatenates the whole payload, which for a multi-GB KV
    /// page means a full-size heap allocation and copy immediately before the
    /// runtime copies again into device memory. Borrowing a mapped range skips
    /// both. The cross-entry overlap that block dedupe exists to exploit is
    /// the shared leading prefix, which the candidate ladder already captures
    /// structurally and more cheaply.
    ///
    /// **Ownership invariant:** a mapped payload holds no `CacheBlobStore`
    /// references. `dedupe_into` must leave it untouched and `release_from`
    /// must not decrement any block ref-count for it, or the blob store will
    /// leak blocks or free them while still referenced.
    Mapped {
        mmap: Arc<Mmap>,
        offset: u64,
    },
}

#[derive(Debug, Clone)]
pub(super) struct CacheBlockRef {
    pub(super) hash: String,
    pub(super) bytes: Arc<Vec<u8>>,
}

impl CacheBlockRef {
    pub(super) fn new(hash: String, bytes: Arc<Vec<u8>>) -> Self {
        Self { hash, bytes }
    }
}

impl CacheBytes {
    pub fn inline(bytes: Vec<u8>) -> Self {
        Self {
            len: bytes.len() as u64,
            repr: CacheBytesRepr::Inline(Arc::new(bytes)),
        }
    }

    pub(super) fn blocks(len: u64, blocks: Vec<CacheBlockRef>) -> Self {
        Self {
            len,
            repr: CacheBytesRepr::Blocks(blocks.into()),
        }
    }

    /// Borrow a validated byte range of a memory-mapped cache file.
    ///
    /// The range is bounds-checked here rather than at read time so a
    /// truncated or corrupt cache file surfaces as an error at load, not as an
    /// out-of-bounds slice during a restore on the serving path.
    pub fn mapped(mmap: Arc<Mmap>, offset: u64, len: u64) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| anyhow!("mapped cache range overflows"))?;
        if end > mmap.len() as u64 {
            return Err(anyhow!(
                "mapped cache range [{offset}, {end}) exceeds mapped file of {} bytes",
                mmap.len()
            ));
        }
        Ok(Self {
            len,
            repr: CacheBytesRepr::Mapped { mmap, offset },
        })
    }

    /// Whether these bytes are backed by the disk tier rather than RAM.
    pub fn is_mapped(&self) -> bool {
        matches!(self.repr, CacheBytesRepr::Mapped { .. })
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_cow(&self) -> Result<Cow<'_, [u8]>> {
        match &self.repr {
            CacheBytesRepr::Inline(bytes) => Ok(Cow::Borrowed(bytes.as_slice())),
            // Borrow, never copy: this is the whole point of the mapped tier.
            CacheBytesRepr::Mapped { mmap, offset } => {
                let start = usize::try_from(*offset)
                    .map_err(|_| anyhow!("mapped cache offset too large for this platform"))?;
                let len = usize::try_from(self.len)
                    .map_err(|_| anyhow!("mapped cache length too large for this platform"))?;
                let end = start
                    .checked_add(len)
                    .ok_or_else(|| anyhow!("mapped cache range overflows"))?;
                let bytes = mmap
                    .get(start..end)
                    .ok_or_else(|| anyhow!("mapped cache range is outside the mapped file"))?;
                Ok(Cow::Borrowed(bytes))
            }
            CacheBytesRepr::Blocks(blocks) => {
                let capacity = usize::try_from(self.len)
                    .map_err(|_| anyhow!("cache payload too large to reconstruct"))?;
                let mut out = Vec::with_capacity(capacity);
                for block in blocks.iter() {
                    out.extend_from_slice(block.bytes.as_slice());
                }
                if out.len() as u64 != self.len {
                    return Err(anyhow!(
                        "cache payload reconstruction length mismatch: expected {} got {}",
                        self.len,
                        out.len()
                    ));
                }
                Ok(Cow::Owned(out))
            }
        }
    }

    pub fn as_cow_timed(&self) -> Result<(Cow<'_, [u8]>, CacheBytesReconstructStats)> {
        let started = Instant::now();
        let blocks = self.block_ref_count();
        let bytes = self.as_cow()?;
        Ok((
            bytes,
            CacheBytesReconstructStats {
                reconstruct_ms: started.elapsed().as_secs_f64() * 1000.0,
                reconstruct_bytes: self.len,
                reconstruct_blocks: blocks,
            },
        ))
    }

    fn block_ref_count(&self) -> usize {
        match &self.repr {
            CacheBytesRepr::Inline(_) | CacheBytesRepr::Mapped { .. } => 0,
            CacheBytesRepr::Blocks(blocks) => blocks.len(),
        }
    }

    /// Blocks owned by this payload in the [`super::CacheBlobStore`].
    ///
    /// Mapped payloads own no blocks, so they must yield nothing here. This is
    /// what makes `release_from` a no-op for them and keeps blob ref-counts
    /// correct across demote/promote.
    pub(super) fn block_hashes(&self) -> impl Iterator<Item = &str> {
        match &self.repr {
            CacheBytesRepr::Inline(_) | CacheBytesRepr::Mapped { .. } => CacheBlockHashIter::Empty,
            CacheBytesRepr::Blocks(blocks) => CacheBlockHashIter::Blocks {
                iter: blocks.iter(),
            },
        }
    }
}

enum CacheBlockHashIter<'a> {
    Empty,
    Blocks {
        iter: std::slice::Iter<'a, CacheBlockRef>,
    },
}

impl<'a> Iterator for CacheBlockHashIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Blocks { iter } => iter.next().map(|block| block.hash.as_str()),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheBytesReconstructStats {
    pub reconstruct_ms: f64,
    pub reconstruct_bytes: u64,
    pub reconstruct_blocks: usize,
}
