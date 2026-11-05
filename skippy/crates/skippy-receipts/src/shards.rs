//! A split GGUF (`name-00001-of-00003.gguf` and its siblings) read as one
//! file: the shards' bytes end to end. Each shard has its own header and
//! the tensors it holds; [`WeightFile::open_shards`] reads them all and
//! places each tensor at its offset in the whole.
//!
//! [`WeightFile::open_shards`]: crate::weights::WeightFile::open_shards

use anyhow::{Result, bail};

use crate::weights::ByteSource;

/// The shards of a split GGUF, end to end.
pub struct Shards {
    /// Each shard with the offset of its first byte in the whole.
    parts: Vec<(u64, Box<dyn ByteSource>)>,
    size: u64,
}

impl Shards {
    pub fn new(shards: Vec<Box<dyn ByteSource>>) -> Self {
        let mut size = 0;
        let parts = shards
            .into_iter()
            .map(|shard| {
                let base = size;
                size += shard.size();
                (base, shard)
            })
            .collect();
        Self { parts, size }
    }

    /// Each shard and the offset of its first byte in the whole.
    pub fn parts(&self) -> impl Iterator<Item = (u64, &dyn ByteSource)> {
        self.parts
            .iter()
            .map(|(base, shard)| (*base, shard.as_ref()))
    }

    /// The pieces of `[offset, offset + len)` in each shard: the shard's
    /// index, the offset in the shard, and the length.
    fn pieces(&self, offset: u64, len: u64) -> Result<Vec<(usize, u64, u64)>> {
        let end = offset.saturating_add(len);
        if end > self.size {
            bail!("bytes {offset}..{end} are past the end of the shards");
        }
        Ok(self
            .parts
            .iter()
            .enumerate()
            .filter_map(|(index, (base, shard))| {
                let start = offset.max(*base);
                let stop = end.min(base + shard.size());
                (start < stop).then(|| (index, start - base, stop - start))
            })
            .collect())
    }

    fn read_with(
        &self,
        offset: u64,
        len: u64,
        read: impl Fn(&dyn ByteSource, u64, u64) -> Result<Vec<u8>>,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(usize::try_from(len)?);
        for (index, at, n) in self.pieces(offset, len)? {
            out.extend(read(self.parts[index].1.as_ref(), at, n)?);
        }
        Ok(out)
    }
}

impl ByteSource for Shards {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.read_with(offset, len, |shard, at, n| shard.read_at(at, n))
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn prefetch(&self, ranges: &[(u64, u64)]) -> Result<()> {
        let mut per_shard = vec![Vec::new(); self.parts.len()];
        for &(offset, len) in ranges {
            for (index, at, n) in self.pieces(offset, len)? {
                per_shard[index].push((at, n));
            }
        }
        for ((_, shard), ranges) in self.parts.iter().zip(per_shard) {
            if !ranges.is_empty() {
                shard.prefetch(&ranges)?;
            }
        }
        Ok(())
    }

    fn read_once(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.read_with(offset, len, |shard, at, n| shard.read_once(at, n))
    }
}

/// The files of the split GGUF whose first shard is `file`, in order, or
/// `None` when `file` is not the first shard of a split.
pub fn shard_files(file: &str) -> Option<Vec<String>> {
    let stem = file.strip_suffix(".gguf")?;
    let (prefix_and_part, total) = stem.rsplit_once("-of-")?;
    let (prefix, part) = prefix_and_part.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(part) || !digits(total) || part != "00001" {
        return None;
    }
    let count: u32 = total.parse().ok().filter(|&n| n > 1)?;
    Some(
        (1..=count)
            .map(|n| format!("{prefix}-{n:05}-of-{total}.gguf"))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::test_gguf::Memory;

    fn shards() -> Shards {
        Shards::new(vec![
            Box::new(Memory((0u8..10).collect())),
            Box::new(Memory((10u8..15).collect())),
            Box::new(Memory((15u8..30).collect())),
        ])
    }

    #[test]
    fn reads_cross_shard_boundaries() {
        let shards = shards();
        assert_eq!(shards.size(), 30);
        assert_eq!(shards.read_at(8, 9).unwrap(), (8u8..17).collect::<Vec<_>>());
        assert_eq!(
            shards.read_at(10, 5).unwrap(),
            (10u8..15).collect::<Vec<_>>()
        );
        assert!(shards.read_at(25, 6).is_err());
    }

    #[test]
    fn the_first_shard_names_its_siblings() {
        assert_eq!(
            shard_files("q/Model-Q4_K_M-00001-of-00003.gguf").unwrap(),
            vec![
                "q/Model-Q4_K_M-00001-of-00003.gguf",
                "q/Model-Q4_K_M-00002-of-00003.gguf",
                "q/Model-Q4_K_M-00003-of-00003.gguf",
            ]
        );
        assert!(shard_files("Model-00002-of-00003.gguf").is_none());
        assert!(shard_files("Model-00001-of-00001.gguf").is_none());
        assert!(shard_files("Model-Q8_0.gguf").is_none());
    }
}
