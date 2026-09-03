use std::{
    borrow::Cow,
    ops::Range,
    sync::{Arc, RwLock},
    time::Instant,
};

use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub struct CacheBytes {
    pub(super) len: u64,
    pub(super) repr: CacheBytesRepr,
}

#[derive(Debug, Clone)]
pub(super) enum CacheBytesRepr {
    Inline(Arc<Vec<u8>>),
    Blocks {
        blocks: Arc<[CacheBlockRef]>,
        /// Original export allocation when every logical block still maps to
        /// one contiguous backing. Restore borrows it without reconstruction.
        contiguous: Option<Arc<Vec<u8>>>,
    },
}

#[derive(Debug, Clone)]
pub(super) struct CacheBlockRef {
    pub(super) hash: String,
    /// Shared indirection lets eviction materialize a surviving deduped block
    /// before releasing its former contiguous backing allocation.
    pub(super) bytes: Arc<RwLock<CacheBlockBytes>>,
}

impl CacheBlockRef {
    pub(super) fn new(hash: String, bytes: Arc<RwLock<CacheBlockBytes>>) -> Self {
        Self { hash, bytes }
    }
}

#[derive(Debug, Clone)]
pub(super) struct CacheBlockBytes {
    pub(super) storage: Arc<Vec<u8>>,
    pub(super) range: Range<usize>,
}

impl CacheBlockBytes {
    pub(super) fn new(storage: Arc<Vec<u8>>, range: Range<usize>) -> Self {
        debug_assert!(range.end <= storage.len());
        Self { storage, range }
    }

    pub(super) fn as_slice(&self) -> &[u8] {
        &self.storage[self.range.clone()]
    }

    pub(super) fn storage_key(&self) -> usize {
        Arc::as_ptr(&self.storage) as usize
    }

    pub(super) fn storage_len(&self) -> usize {
        self.storage.len()
    }
}

impl CacheBytes {
    pub fn inline(bytes: Vec<u8>) -> Self {
        Self {
            len: bytes.len() as u64,
            repr: CacheBytesRepr::Inline(Arc::new(bytes)),
        }
    }

    pub(super) fn blocks(
        len: u64,
        blocks: Vec<CacheBlockRef>,
        contiguous: Option<Arc<Vec<u8>>>,
    ) -> Self {
        Self {
            len,
            repr: CacheBytesRepr::Blocks {
                blocks: blocks.into(),
                contiguous,
            },
        }
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
            CacheBytesRepr::Blocks { blocks, contiguous } => {
                let capacity = usize::try_from(self.len)
                    .map_err(|_| anyhow!("cache payload too large to reconstruct"))?;
                if let Some(bytes) = contiguous {
                    return Ok(Cow::Borrowed(bytes.as_slice()));
                }
                let mut out = Vec::with_capacity(capacity);
                for block in blocks.iter() {
                    let bytes = block
                        .bytes
                        .read()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    out.extend_from_slice(bytes.as_slice());
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
        let reconstructed = matches!(bytes, Cow::Owned(_));
        Ok((
            bytes,
            CacheBytesReconstructStats {
                reconstruct_ms: started.elapsed().as_secs_f64() * 1000.0,
                reconstruct_bytes: if reconstructed { self.len } else { 0 },
                reconstruct_blocks: if reconstructed { blocks } else { 0 },
            },
        ))
    }

    fn block_ref_count(&self) -> usize {
        match &self.repr {
            CacheBytesRepr::Inline(_) => 0,
            CacheBytesRepr::Blocks { blocks, .. } => blocks.len(),
        }
    }

    pub(super) fn block_hashes(&self) -> impl Iterator<Item = &str> {
        match &self.repr {
            CacheBytesRepr::Inline(_) => CacheBlockHashIter::Empty,
            CacheBytesRepr::Blocks { blocks, .. } => CacheBlockHashIter::Blocks {
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
